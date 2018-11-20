import PropTypes from 'prop-types';

export default PropTypes.shape({
  loading: PropTypes.bool.isRequired,
  loadContextFromUrl: PropTypes.bool.isRequired,
  query: PropTypes.string.isRequired,
  context: PropTypes.string.isRequired,
  contextUrl: PropTypes.string.isRequired,
  errorCode: PropTypes.number.isRequired,
});
