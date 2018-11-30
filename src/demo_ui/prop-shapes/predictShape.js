import PropTypes from 'prop-types';

export default PropTypes.shape({
  numPredictions: PropTypes.number.isRequired,
  bestAnswer: PropTypes.string.isRequired,
  data: PropTypes.arrayOf(PropTypes.shape({
    contextTokens: PropTypes.arrayOf(PropTypes.string).isRequired,
    answerText: PropTypes.string.isRequired,
    answerStart: PropTypes.number.isRequired,
    answerEnd: PropTypes.number.isRequired,
    startProb: PropTypes.arrayOf(PropTypes.number).isRequired,
    endProb: PropTypes.arrayOf(PropTypes.number).isRequired,
  })).isRequired,
});
