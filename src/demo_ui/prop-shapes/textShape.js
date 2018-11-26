import PropTypes from 'prop-types';

export default PropTypes.shape({
  loading: PropTypes.bool.isRequired,
  loadExample: PropTypes.bool.isRequired,
  query: PropTypes.string.isRequired,
  context: PropTypes.string.isRequired,
});
