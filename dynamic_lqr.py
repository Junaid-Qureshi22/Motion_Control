def compute_lqr_gains_dynamic(V, b):
    m = 550
    Cf = 340
    Cr = 340
    lf = b/2
    lr = b/2
    Iz = 3200
    A = [[-2*(Cf+Cr)/(m*V), -V-2*(lf*Cf-lr*Cr)/(m*V**2)], [-2*(lf*Cf-lr*Cr)/(Iz*V), 2*(lf*lf*Cf+lr*lr*Cr)/(Iz*V)]]
    B = [[2 * Cf / m], [2 *lf * Cf / Iz]]
    Q = [[1, 0], [0, 1]]
    R = [[10]]
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.dot(np.linalg.inv(R),(np.dot(np.transpose(B),X)))
    K1 = K[0, 0]
    K2 = K[0, 1]
    return K1, K2




    elif (type == "lqr_dynamic_f"):
	    u_k = feedforward + K1 * (V * np.sin(proj_heading) - utils.dy(V, theta[i], beta, model)) + K2 *  (utils.dtheta(V, phi[i], beta, wheelbase, model) - curvature * V)



      elif (type == "lqr_dynamic" or type == "lqr_dynamic_f"):
          K1, K2 = utils.compute_lqr_gains_dynamic(V_init, wheelbase)
          print("K1 = {}".format(K1))
          print("K2 = {}".format(K2))

          # -- lqr_dynamic: dynamic linear quadratic regulator
          # -- lqr_dynamic_f: linear quadratic regulator with feedforward action
