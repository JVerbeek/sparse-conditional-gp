import gpflow as gpf
import tensorflow as tf
import numpy as np
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from gpflow.inducing_variables import InducingPoints
from gpflow.models.util import inducingpoint_wrapper


class ConditionalMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    """GPFlow implementation of the AT-GP model proposed by Cao et al. (https://ojs.aaai.org/index.php/AAAI/article/view/7682/7543)"""
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=1, conditioning_index=1):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = data
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_index = conditioning_index

    def conditional_likelihood(self, *args, **kwargs) -> tf.Tensor:
        def get_condition_number(M, name=""):
            """Debug tool for if matrices turn non-singular"""
            s = tf.linalg.svd(M, compute_uv=False)
            tf.print(f"condition number {name}", s[0] / s[-1], s[0], s[-1])
            return

        Xs, Ys = self.data
        Kall = self.kernel(Xs)

        # Determine what is source and what is target (currently use only one target var)
        As = Ys[:, 1] != self.conditioning_index
        Bs = Ys[:, 1] == self.conditioning_index

        Ax, Ay = tf.reshape(Xs[:, 0][As], (-1, 1)), tf.reshape(Ys[:, 0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:, 0][Bs], (-1, 1)), tf.reshape(Ys[:, 0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])

        # Compute all sub-matrices by collecting the kernel values for source and target.
        Kaa = tf.gather(
            tf.gather(Kall, indices_A, axis=0), indices_A, axis=1
        ) + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Ax)))
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kbb = tf.gather(
            tf.gather(Kall, indices_B, axis=0), indices_B, axis=1
        ) + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx)))
        Kba = tf.transpose(Kab)

        Laa = tf.linalg.cholesky(Kaa)
        A = tf.linalg.triangular_solve(
            Laa, Ay
        )  # Laa * x = Ay, so then x = inv(Laa) * Ay
        V = tf.linalg.triangular_solve(
            Laa, Kab
        )  # Laa * x = Kab, so then x = inv(Laa) * Kab

        mu_t = tf.matmul(V, A, transpose_a=True)
        # Math:
        # transpose(inv(Laa) * Kab) = Kba * inv(Laa).T
        # mu_t = Kba * inv(Laa).T * inv(Laa) * Ay
        #        <=> Kba * inv(LaaLaa.T) * Ay

        C_t = Kbb - tf.matmul(V, V, transpose_a=True)
        # Math:
        # We want to compute Kbb - Kba inv(Kaa) Kab, so since
        # Kba inv(Kaa) Kab <=> Kba * inv(Laa).T * inv(Laa) * Kab <=> Kba * (Laa * Laa.T) * Kab

        L_t = tf.linalg.cholesky(C_t)

        # product of cholesky diagonals is determinant, therefore sum of log cholesky is also determinant
        logdet_t = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))

        delta = By - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta)
        # Math
        # L_t * x = delta, then alpha_t = inv(L_t)*(By - mu_t)

        n_target = tf.cast(
            tf.shape(Bx)[0], Bx.dtype
        )  # We compute for the target (given source)
        quad_part = tf.matmul(alpha_t, alpha_t, transpose_a=True)

        lml = -0.5 * (
            logdet_t + quad_part + n_target * tf.cast(tf.math.log(2 * np.pi), Ax.dtype)
        )
        return tf.squeeze(lml)

    def maximum_log_likelihood_objective(self):
        return self.conditional_likelihood()

    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        r"""
        Allegedly, the GP prediction stays the same, so instead of creating an inference shaped footgun, use GPFlow methods.
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        Xs, Ys = self.data
        Kall = self.kernel(Xs)
        err = (Ys - self.mean_function(Xs))[:, 0][:, None]

        As = Ys[:, 1] != self.conditioning_index
        Bs = Ys[:, 1] == self.conditioning_index
        err_As = err[As]
        err_Bs = err[Bs]
        err_reorder = tf.concat((err_As, err_Bs), axis=0)

        knn = self.kernel(Xnew, full_cov=full_cov)

        Ax, Ay = tf.reshape(Xs[:, 0][As], (-1, 1)), tf.reshape(Ys[:, 0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:, 0][Bs], (-1, 1)), tf.reshape(Ys[:, 0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])

        X_ordered = tf.gather(Xs, tf.concat([indices_A, indices_B], axis=0))

        # Compute exact kernel parts
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1)
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1)
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kba = tf.transpose(Kab)

        Kmm = tf.concat((tf.concat((Kaa, Kba), 0), tf.concat((Kab, Kbb), 0)), 1)
        noise = tf.squeeze(
            tf.linalg.diag(
                tf.concat(
                    (
                        self.likelihood.source.variance_at(Ax),
                        self.likelihood.target.variance_at(Bx),
                    ),
                    0,
                )
            )
        )
        kmm_plus_s = Kmm + tf.linalg.diag(noise)
    
        # Construct Kmn
        Knm = self.kernel(X_ordered, Xnew)
        Lkmm = tf.linalg.cholesky(kmm_plus_s)
        KnminvLmm = tf.linalg.triangular_solve(Lkmm, Knm)  # Lkmm-1 Knm
        Lkmmy = tf.linalg.triangular_solve(Lkmm, err_reorder)  # Lkmm-1 err

        cond = tf.transpose(KnminvLmm) @ KnminvLmm
        f_mean_zero = tf.transpose(KnminvLmm) @ Lkmmy
        fvar = tf.expand_dims(tf.linalg.diag_part(knn - cond), 1)
        f_mean = (f_mean_zero + self.mean_function(Xnew))
        return f_mean, fvar


class SparseCMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data,
        kernel,
        likelihood,
        mean_function=None,
        num_latent_gps=1,
        inducing_variable=[],
        exact_target=False,
        jitter=1e-6,
        conditioning_index=1
    ):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = data
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_variable = inducingpoint_wrapper(inducing_variable) #InducingPoints(inducing_variable[:, 0].reshape(-1, 1))
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_indices = [0]
        self.opt_logs = []
        self.exact_target = exact_target
        self.jitter = tf.cast(jitter, tf.float64)
        self.conditioning_index = conditioning_index

    def conditional_likelihood(
        self, *args, decompose_likelihood=False, **kwargs
    ) -> tf.Tensor:
        Xs, Ys = self.data
        Xind = self.inducing_variable.Z

        # Determine what is source and what is target (currently use only one target var)
        As = Ys[:, 1] != self.conditioning_index
        Bs = Ys[:, 1] == self.conditioning_index

        Ax, Ay = tf.reshape(Xs[:, 0][As], (-1, 1)), tf.reshape(Ys[:, 0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:, 0][Bs], (-1, 1)), tf.reshape(Ys[:, 0][Bs], (-1, 1))

        inducing_variable = self.inducing_variable.Z
        Kmm = self.kernel(
            inducing_variable
        )  + tf.eye(inducing_variable.shape[0], dtype=tf.float64) * self.jitter
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kam = tf.transpose(Kma)

        # Steps to avoid explicit inverse of Kmm
        L_Kmm = tf.linalg.cholesky(Kmm)  # [1]
        Lmm_inv_kma = tf.linalg.triangular_solve(
            L_Kmm, Kma
        )  # L_Kmm x = Kma, x = L_Kmm^-1 Kma
        Lmm_inv_kmb = tf.linalg.triangular_solve(
            L_Kmm, Kmb
        )  # L_Kmm x = Kmb, x = L_Kmm^-1 Kmb

        # Compute diagonals
        diag_Qtt = tf.reduce_sum(Lmm_inv_kmb ** 2, axis=0) 
        diag_Kall = self.kernel.K_diag(Xs[:,1][:,None])
        diag_Kss = diag_Kall[As]
        diag_Ktt = diag_Kall[Bs]
        sigma_t = tf.squeeze(self.likelihood.target.variance_at(Bx))
        D_t = (diag_Ktt - diag_Qtt) + sigma_t

        diag_Qss = tf.reduce_sum(Lmm_inv_kma ** 2, axis=0)
        sigma_s = tf.squeeze(self.likelihood.source.variance_at(Ax))

        D_s = (diag_Kss - diag_Qss) + sigma_s
        W = Kmm + (Kma * (1.0/D_s)[None,:]) @ Kam + tf.eye(inducing_variable.shape[0], dtype=tf.float64) * self.jitter
        Lw = tf.linalg.cholesky(W) 

        u = Ay / D_s[:,None]
        mu_t = tf.transpose(Kmb) @ tf.linalg.cholesky_solve(
            Lw, Kma @ u
        )  # Kbm W^1 Kma Ds^-1 Ay

        BmWmt = tf.transpose(Kmb) @ tf.linalg.cholesky_solve(
            Lw, Kmb
        )  # LwLw^T x = Kmb, x = W^1 Kbm, res: Kmb W^-1 Kbm

        if not self.exact_target:  # Then we can use the Woodbury inverse!
            C_t = BmWmt + tf.linalg.diag(D_t) 
            delta = By - mu_t
            middle = W + (Kmb * (1.0 /D_t)[None,:]) @ tf.transpose(Kmb)
            m_chol = tf.linalg.cholesky(middle)
            a = delta /D_t[:,None] # D_t_inv u 
            b = tf.linalg.triangular_solve(m_chol, Kmb @ a)  
            quad = tf.matmul(delta, a, transpose_a=True) - tf.matmul(b, b, transpose_a=True)
            n_target = tf.cast(tf.shape(Bx)[0], Bx.dtype)
            Lw_inv_Kmb = tf.linalg.triangular_solve(Lw, Kmb)  # O(m^2 * n_b)
            inner = tf.eye(Lw_inv_Kmb.shape[0], dtype=tf.float64) + (Lw_inv_Kmb * (1.0/ D_t[None,:])) @ tf.transpose(Lw_inv_Kmb)
            logdet_inner = tf.linalg.logdet(inner)
            logdet_t = tf.reduce_sum(tf.math.log(D_t)) + logdet_inner 
            lml = -0.5 * (
                logdet_t
                + quad
                + n_target * tf.cast(tf.math.log(2 * tf.constant(np.pi)), Bx.dtype)
            )

        else:
            Kbb = self.kernel(Xs[Bs])
            Qbb = tf.matmul(Lmm_inv_kmb, Lmm_inv_kmb, transpose_a=True)
            C_t = Kbb - Qbb + BmWmt + tf.linalg.diag(D_t)
            delta = By - mu_t
            L_t = tf.linalg.cholesky(C_t)
            alpha_t = tf.linalg.triangular_solve(L_t, delta) 
            n_target = tf.cast(tf.shape(Bx)[0], Bx.dtype)
            logdet_t = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
            quad = tf.matmul(alpha_t, alpha_t, transpose_a=True)
            lml = -0.5 * (
                logdet_t
                + quad
                + n_target * tf.cast(tf.math.log(2 * tf.constant(np.pi)), Bx.dtype)
            )

        return tf.squeeze(lml)

    def maximum_log_likelihood_objective(self):
        return self.conditional_likelihood()

    def predict_f(self, Xnew, full_cov=False, **kwargs):
        Xs, Ys = self.data

        err = (Ys - self.mean_function(Xs))[:, 0][:, None]

        As = Ys[:, 1] != self.conditioning_index
        Bs = Ys[:, 1] == self.conditioning_index

        err_reorder = tf.concat((err[As], err[Bs]), axis=0)

        Ax, Ay = tf.reshape(Xs[:, 0][As], (-1, 1)), tf.reshape(Ys[:, 0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:, 0][Bs], (-1, 1)), tf.reshape(Ys[:, 0][Bs], (-1, 1))

        inducing_variable = self.inducing_variable.Z
        M = inducing_variable.shape[0]
        T = tf.shape(Bx)[0]

        Kmm = self.kernel(inducing_variable) + tf.eye(M, dtype=tf.float64) * self.jitter
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kaa = self.kernel(Xs[As])
        Kbb = self.kernel(Xs[Bs])

        L_Kmm = tf.linalg.cholesky(Kmm)
        Lmm_inv_kma = tf.linalg.triangular_solve(L_Kmm, Kma) 
        Lmm_inv_kmb = tf.linalg.triangular_solve(L_Kmm, Kmb) 

        Qaa = tf.matmul(Lmm_inv_kma, Lmm_inv_kma, transpose_a=True) 
        Qbb = tf.matmul(Lmm_inv_kmb, Lmm_inv_kmb, transpose_a=True)  
        Qab = tf.matmul(Lmm_inv_kma, Lmm_inv_kmb, transpose_a=True)  
        Qba = tf.transpose(Qab)                                    

        sigma_s = tf.squeeze(self.likelihood.source.variance_at(Ax))
        sigma_t = tf.squeeze(self.likelihood.target.variance_at(Bx))
        diag_D_a = (tf.linalg.diag_part(Kaa) - tf.linalg.diag_part(Qaa)) + sigma_s  
        diag_D_b = (tf.linalg.diag_part(Kbb) - tf.linalg.diag_part(Qbb)) + sigma_t  

        D_a_inv = tf.linalg.diag(1.0 / diag_D_a)

        # Blockwise inversion of K_fic
        W = Kmm + Kma @ D_a_inv @ tf.transpose(Kma)
        Lw = tf.linalg.cholesky(W) 
        half = tf.linalg.triangular_solve(Lw, Kma @ D_a_inv)  
        A_inv = D_a_inv - tf.matmul(half, half, transpose_a=True)  
        QbaKaa_inv = Qba @ A_inv   

        BmWmt = tf.transpose(Kmb) @ tf.linalg.cholesky_solve(
            Lw, Kmb
        )  # LwLw^T x = Kmb, x = W^1 Kmb
        C_t = BmWmt + tf.linalg.diag(diag_D_b) 
        D_t_inv = tf.linalg.diag(1.0 / diag_D_b)
        middle = W + Kmb @ D_t_inv @ tf.transpose(Kmb)
        m_chol = tf.linalg.cholesky(middle)
        middle_inv = tf.linalg.triangular_solve(m_chol, Kmb @ D_t_inv)
        C_t_inv = D_t_inv - tf.matmul(middle_inv, middle_inv, transpose_a=True)
        
        # Compute blockwise inversion blocks
        C_block = C_t_inv 
        B_block = -tf.matmul(C_t_inv, QbaKaa_inv)                        
        A_block = A_inv + tf.transpose(QbaKaa_inv) @ C_block @ QbaKaa_inv      
        K_fic_inv = tf.concat([
            tf.concat([A_block, tf.transpose(B_block)], axis=1),
            tf.concat([B_block, C_block], axis=1),
        ], axis=0) 

        # Put together cross-covariance between new and training data
        Kmnew = self.kernel(inducing_variable, Xnew)                               
        Lmm_inv_kmnew = tf.linalg.triangular_solve(L_Kmm, Kmnew)            
        Qan = tf.matmul(Lmm_inv_kma,  Lmm_inv_kmnew, transpose_a=True)            
        Qbn = tf.matmul(Lmm_inv_kmb,  Lmm_inv_kmnew, transpose_a=True)          
        Kfn = tf.concat([Qan, Qbn], axis=0)                          

        # Mean: Qun (K_fic)^-1 y (FITC)
        c = tf.transpose(Kfn) @ K_fic_inv 
        f_mean_zero = c @ err_reorder      
        f_mean = f_mean_zero + self.mean_function(Xnew[:, 0][:, None])

        # Cov: Knn - Qn Kfic^-1 Qn^T 
        knn = self.kernel(Xnew, full_cov=False)                                 
        diagonal_variance = tf.reduce_sum(Lmm_inv_kmnew ** 2, axis=0)     
        f_var = tf.expand_dims(knn - tf.linalg.diag_part(c @ Kfn), 1)               
        return f_mean, f_var
