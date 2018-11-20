import PropTypes from 'prop-types';

export default PropTypes.shape({
  query: PropTypes.string.isRequired,
  numAnswers: PropTypes.number.isRequired,
  contextTokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  queryTokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  answerTexts: PropTypes.arrayOf(PropTypes.string).isRequired,
  answerStarts: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
  answerEnds: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
  startProb: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
  endProb: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
});
