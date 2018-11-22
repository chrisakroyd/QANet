import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import { push } from 'react-router-redux';

import { SET_CONTEXT_TEXT, SET_CONTEXT_URL_TEXT, SET_QUERY_TEXT } from '../constants/actions';
import { getPrediction } from '../actions/compositeActions';
import { setInputText, setURLFlag } from '../actions/textActions';
import Demo from '../components/Demo/Demo';


const mapStateToProps = state =>
  ({
    text: state.text,
    predictions: state.predictions,
  });

const mapDispatchToProps = dispatch => ({
  predict: () => {
    console.log('predict');
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
  setContextUrlText: (text) => {
    dispatch(setInputText(text, SET_CONTEXT_URL_TEXT));
  },
  setContextUrlFlag: (flag) => {
    dispatch(setURLFlag(flag));
  },
});

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(Demo));
