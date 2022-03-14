extern crate hello;

use smartcore::dataset::boston;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::model_selection::train_test_split;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
use smartcore::metrics::mean_squared_error;

fn main() {
  // 自作クレート
  let word:&str = "World";
  println!("{:?}", hello::hello_world(word));

  // 機械学習
  // データセットのダウンロード
  let boston_data = boston::load_dataset();

  // 特徴量を arrayデータへ変換
  let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
  );

  // 目的変数
  let y = boston_data.target;

  // 評価データと訓練データに分割
  let (x_train, x_test, y_train, y_test) = train_test_split(
                                                    &x,    // training data
                                                    &y,    // target
                                                    0.2,   // test_size
                                                    false, // shuffle
                                                  );
  // 学習
  let model = LinearRegression::fit(
                            &x_train, 
                            &y_train,
                            LinearRegressionParameters::default().
                              with_solver(LinearRegressionSolverName::QR)
                            ).unwrap();

  // 推論
  let y_pred = model.predict(&x_test).unwrap();

  // 評価
  let metrics = mean_squared_error(&y_test, &y_pred);

  // 評価結果を出力
  println!("mse:{}", metrics);
}
