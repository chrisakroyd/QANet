const common = require('./webpack.common.js');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const merge = require('webpack-merge');

module.exports = merge(common, {
  devtool: 'cheap-module-eval-source-map',
  mode: 'development',
  devServer: {
    contentBase: './dist',
    historyApiFallback: true,
  },
  plugins: [new BundleAnalyzerPlugin({
    analyzerMode: 'server',
    analyzerHost: '127.0.0.1',
    // Port
    analyzerPort: 8888,
    defaultSizes: 'parsed',
  })],
});
