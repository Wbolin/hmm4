// Baum-Welch Algorithm Implementation
public class Main {
	private static double[][] transition, emission, initial;
	private static int[] emissionSequence;
	private static Kattio io = new Kattio(System.in, System.out);
	private static double[] C;
	private static double[][] gamma;
	private static double[][][] gamma_;
	private static double[][] alpha;
	private static double[][] beta;

	public static void main(String[] args) {
		transition = handleInput();
		emission = handleInput();
		initial = handleInput();
		emissionSequence = handleSequenceInput();
		int T = emissionSequence.length;
		C = new double[T];

		// Set initial state probability distribution as a one dimensional
		// vector instead of a matrix.
		double[] pi = new double[initial[0].length];
		for (int i = 0; i < pi.length; i++)
			pi[i] = initial[0][i];

		int maxIters = 500;
		int iters = 0;
		double oldLogProb = -1000000.0;
		double logProb = 0.0;

		while (true) {
			// The a-pass
			alpha = alphaPass(emissionSequence, pi, transition, emission);
			// The ß-pass
			beta = betaPass(emissionSequence, pi, transition, emission);
			// Compute ?t(i, j) and ?t(i)
			computeGamma(emissionSequence, transition, emission);
			// Re-estimate A, B and p
			reEstimate(emissionSequence, pi, transition, emission);

			// Compute log[P(O | ?)]
			logProb = 0.0;
			for (int i = 0; i < T; i++)
				logProb += log(C[i], 10);
			logProb = -logProb;

			// To iterate or not to iterate, that is the question. . .
			iters++;
			if ((iters < maxIters) && (logProb > oldLogProb))
				oldLogProb = logProb;
			else
				break;
		}

		// Output ? = (p, A, B)
		printMatrix(transition);
		printMatrix(emission);

		// Close IO-stream
		io.close();
	}

	private static double log(double x, int base) {
		return (Math.log(x) / Math.log(base));
	}

	private static void reEstimate(int[] O, double[] pi, double[][] A,
			double[][] B) {

		int N = transition.length;
		int T = O.length;
		int M = emission[0].length;

		// re-estimate p
		for (int i = 0; i < N; i++) {
			pi[i] = gamma[0][i];
		}

		// re-estimate A
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				double numer = 0.0;
				double denom = 0.0;
				for (int t = 0; t < T - 1; t++) {
					numer += gamma_[t][i][j];
					denom += gamma[t][i];
				}
				A[i][j] = numer / denom;
			}
		}

		// re-estimate B
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				double numer = 0.0;
				double denom = 0.0;
				for (int t = 0; t < T - 1; t++) {
					if (O[t] == j)
						numer += gamma[t][i];
					denom += gamma[t][i];
				}
				B[i][j] = numer / denom;
			}
		}

		for (int i = 0; i < pi.length; i++)
			initial[0][i] = pi[i];

		transition = A;
		emission = B;

	}

	private static void computeGamma(int[] O, double[][] A, double[][] B) {
		int N = transition.length;
		int T = O.length;

		gamma = new double[T][N];
		gamma_ = new double[T][N][N];

		for (int t = 0; t < T - 1; t++) {
			double denom = 0.0;
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					denom += alpha[t][i] * A[i][j] * B[j][O[t + 1]]
							* beta[t + 1][j];
				}
			}

			for (int i = 0; i < N; i++) {
				gamma[t][i] = 0.0;
				for (int j = 0; j < N; j++) {
					gamma_[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j])
							/ denom;
					gamma[t][i] += gamma_[t][i][j];
				}
			}
		}
	}

	private static double[][] betaPass(int[] O, double[] pi, double[][] A,
			double[][] B) {
		int N = transition.length;
		int T = O.length;

		double[][] beta = new double[T][N];

		// Scale beta.
		for (int i = 0; i < N; i++)
			beta[T - 1][i] = C[T - 1];

		// Beta-pass
		for (int t = T - 2; t >= 0; t--) {
			for (int i = 0; i < N; i++) {
				beta[t][i] = 0.0;
				for (int j = 0; j < N; j++)
					beta[t][i] = beta[t][i]
							+ (A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]);
				// Scale beta with same factor as alpha
				beta[t][i] = C[t] * beta[t][i];
			}
		}

		return beta;
	}

	private static double[][] alphaPass(int[] O, double[] pi, double[][] A,
			double[][] B) {
		// Initialization
		int N = transition.length;
		int T = O.length;

		double[][] alpha = new double[T][N];

		// Compute alpha[0][i]
		C[0] = 0.0;
		for (int i = 0; i < N; i++) {
			alpha[0][i] = pi[i] * B[i][O[0]];
			C[0] += alpha[0][i];
		}

		// Scale the alpha[0][i]
		C[0] = 1.0 / C[0];
		for (int i = 0; i < N; i++)
			alpha[0][i] = C[0] * alpha[0][i];

		// Compute alpha[t][i]
		for (int t = 1; t < T; t++) {
			C[t] = 0.0;
			for (int i = 0; i < N; i++) {
				alpha[t][i] = 0.0;
				for (int j = 0; j < N; j++)
					alpha[t][i] = alpha[t][i] + (alpha[t - 1][j] * A[j][i]);
				alpha[t][i] = alpha[t][i] * B[i][O[t]];
				C[t] = C[t] + alpha[t][i];
			}
			// Scale alpha[t][i]
			C[t] = 1.0 / C[t];
			for (int i = 0; i < N; i++)
				alpha[t][i] = C[t] * alpha[t][i];
		}
		return alpha;
	}

	private static int[] viterbi(int[] S, double[] pi, int[] Y, double[][] A,
			double[][] B) {
		int K = transition.length;
		int T = emissionSequence.length;
		double[][] T_1 = new double[K][T];
		int[][] T_2 = new int[K][T];

		for (int i = 0; i < K; i++) {
			T_1[i][0] = (pi[i] * B[i][Y[0]]);
			T_2[i][0] = 0;
		}

		for (int i = 1; i < T; i++) {
			for (int j = 0; j < K; j++) {
				double maxValue = 0;
				int maxArg = 0;
				for (int k = 0; k < K; k++) {
					double value = T_1[k][i - 1] * A[k][j] * B[j][Y[i]];
					if (value > maxValue) {
						maxValue = value;
						maxArg = k;
					}
				}
				T_1[j][i] = maxValue;
				T_2[j][i] = maxArg;
			}
		}

		int[] Z = new int[T];
		int[] X = new int[T];
		double value = 0;

		for (int k = 0; k < K; k++) {
			if (T_1[k][T - 1] > value)
				Z[T - 1] = k;
		}

		X[T - 1] = S[Z[T - 1]];

		for (int i = T - 1; i > 0; i--) {
			Z[i - 1] = T_2[Z[i]][i];
			X[i - 1] = S[Z[i - 1]];
		}

		return X;

	}

	private static double[][] handleInput() {
		int rows = io.getInt();
		int cols = io.getInt();
		double[][] matrix = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = io.getDouble();
			}
		}
		return matrix;
	}

	private static int[] handleSequenceInput() {
		int rows = io.getInt();
		int[] sequence = new int[rows];
		for (int i = 0; i < rows; i++)
			sequence[i] = io.getInt();
		return sequence;
	}

	private static void printMatrix(double[][] matrix) {
		int rows = 0;
		int cols = 0;
		String line = "";
		for (double[] row : matrix) {
			rows++;
			for (double j : row) {
				cols++;
				line = line + (j + " ");
			}
		}
		System.out.println(matrix.length + " " + matrix[0].length + " " + line);
	}

	// return C = A * B
	public static double[][] multiply(double[][] A, double[][] B) {
		int mA = A.length;
		int nA = A[0].length;
		int mB = B.length;
		int nB = B[0].length;
		if (nA != mB)
			throw new RuntimeException("Illegal matrix dimensions.");
		double[][] C = new double[mA][nB];
		for (int i = 0; i < mA; i++)
			for (int j = 0; j < nB; j++)
				for (int k = 0; k < nA; k++)
					C[i][j] += (A[i][k] * B[k][j]);
		return C;
	}
}