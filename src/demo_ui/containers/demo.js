import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import { push } from 'react-router-redux';

import { SET_CONTEXT_TEXT, SET_QUERY_TEXT } from '../constants/actions';
import { getPrediction, getExample } from '../actions/compositeActions';
import { setInputText, enterTextManually } from '../actions/textActions';
import Demo from '../components/Demo/Demo';


const mapStateToProps = state =>
  ({
    text: state.text,
    predictions: state.predictions,
  });

const mapDispatchToProps = dispatch => ({
  predict: () => {
    dispatch(getPrediction());
    dispatch(push('/results'));
  },
  returnHome: () => {
    dispatch(push('/'));
  },
  setQueryText: (text) => {
    dispatch(setInputText(text, SET_QUERY_TEXT));
  },
  setContextText: (text) => {
    dispatch(setInputText(text, SET_CONTEXT_TEXT));
  },
  enterText: (flag) => {
    dispatch(enterTextManually(flag));
  },
  loadExample: () => {
    dispatch(getExample());
  },
});

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Demo));
