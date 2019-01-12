import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import { push } from 'react-router-redux';

import { runPrediction, getExample, setContextText, setQueryText } from '../actions/compositeActions';
import { clearError } from '../actions/predictActions';
import { enterTextManually } from '../actions/textActions';
import Demo from '../components/Demo/Demo';


const mapStateToProps = state =>
  ({
    text: state.text,
    predictions: state.predictions,
  });

const mapDispatchToProps = dispatch => ({
  predict: () => {
    dispatch(runPrediction());
  },
  returnHome: () => {
    dispatch(clearError());
    dispatch(push('/'));
  },
  setQueryText: (text) => {
    dispatch(setQueryText(text));
  },
  setContextText: (text) => {
    dispatch(setContextText(text));
  },
  enterText: (flag) => {
    dispatch(enterTextManually(flag));
  },
  loadExample: () => {
    dispatch(getExample());
  },
});

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Demo));
