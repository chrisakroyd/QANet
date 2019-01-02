const merge = require('webpack-merge');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  devtool: 'source-map',
  mode: 'production',
  optimization: {
    minimizer: [new TerserPlugin()],
  },
  plugins: [
    new CleanWebpackPlugin(['dist'], { verbose: true }),
  ],
});
