// @TODO Hook this file into defaults.json in /data/
const debug = true;
const devUrl = 'http://localhost:8080';
// @TODO: Hook this port into defaults.json.
const prodUrl = 'http://localhost:5000';

module.exports = {
  debug,
  siteUrl: debug ? devUrl : prodUrl,
};
