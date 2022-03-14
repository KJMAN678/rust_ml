use smartcore::dataset::boston;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
use smartcore::metrics::mean_squared_error;

fn main() {

  let boston_data = boston::load_dataset();

  let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
  );

  let y = boston_data.target;

  let (x_train, x_test, y_train, y_test) = train_test_split(
                                                    &x,    // training data
                                                    &y,    // target
                                                    0.2,   // test_size
                                                    false, // shuffle
                                                  );

  let model = LinearRegression::fit(
                            &x_train, 
                            &y_train,
                            LinearRegressionParameters::default().
                              with_solver(LinearRegressionSolverName::QR)
                            ).unwrap();

  let y_pred = model.predict(&x_test).unwrap();

  let metrics = mean_squared_error(&y_test, &y_pred);

  println!("mse:{}", metrics);

}
