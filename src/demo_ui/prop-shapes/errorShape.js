import PropTypes from 'prop-types';

export default PropTypes.shape({
  errorCode: PropTypes.number.isRequired,
  errorMessage: PropTypes.string.isRequired,
  parameters: PropTypes.shape({
    context: PropTypes.string.isRequired,
    query: PropTypes.string.isRequired,
  }),
});
