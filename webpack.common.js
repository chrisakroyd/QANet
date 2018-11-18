const CopyWebpackPlugin = require('copy-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');

const outputDirectory = path.resolve('./dist/');

const htmlWebpackPluginConfig = new HtmlWebpackPlugin({
  template: './src/demo_ui/index.html',
  filename: 'index.html',
  inject: 'body',
});


module.exports = {
  entry: './src/demo_ui/index.jsx',
  output: {
    path: outputDirectory,
    filename: 'bundle.js',
  },
  module: {
    loaders: [
    ],
    rules: [
      { test: /\.(js|jsx)$/,
        loader: 'babel-loader',
        exclude: /node_modules/,
        options: {
          presets: ['es2015', 'react', 'stage-2'],
        },
      },
      {
        test: /\.scss$/,
        use: [{
          loader: 'style-loader',
        }, {
          loader: 'css-loader',
        }, {
          loader: 'sass-loader',
          options: {
            includePaths: ['absolute/path/a', 'absolute/path/b'],
          },
        }],
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.jsx'],
  },
  plugins: [
    htmlWebpackPluginConfig,
    new CopyWebpackPlugin([
      { from: 'src/frontend/resources/images', to: `${outputDirectory}/images` }, // Should copy all images used in the actual site.
    ]),
  ],
};
