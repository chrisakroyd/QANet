const merge = require('webpack-merge');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const UglifyJSPlugin = require('uglifyjs-webpack-plugin');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  devtool: 'source-map',
  mode: 'production',
  plugins: [
    new CleanWebpackPlugin(['dist'], { verbose: true }),
    new UglifyJSPlugin({
      sourceMap: true,
    }),
  ],
});
