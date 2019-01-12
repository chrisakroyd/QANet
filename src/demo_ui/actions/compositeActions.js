import axios from 'axios';
import { push } from 'react-router-redux';

import config from '../config';

import { SET_CONTEXT_TEXT, SET_QUERY_TEXT } from '../constants/actions';
import { loadExample, loadExampleSuccess, loadExampleFailure, setInputText } from './textActions';
import { predict, predictSuccess, predictFailure, clearError } from './predictActions';

const demoUrl = `http://localhost:${config.demoPort}`;

// Simple test to determine if the user has made changes to the context or query that could
// constitute a solution to an error generated on the back-end (e.g. as a result of tokenisation)
function isErrorFixed(error, text, key) {
  let isFixed = false;
  if (error !== null) {
    const errorText = error.parameters[key];
    isFixed = text !== errorText;
  }
  return isFixed;
}

export function runPrediction() {
  return (dispatch, getState) => {
    const { text } = getState();

    dispatch(predict());
    return axios.post(`${demoUrl}/api/v1/model/predict`, { context: text.context, query: text.query })
      .then(res => dispatch(predictSuccess(res.data)))
      .then(() => dispatch(push('/results')))
      .catch(err => dispatch(predictFailure(err)));
  };
}

export function getExample() {
  return (dispatch) => {
    dispatch(loadExample());
    return axios.get(`${demoUrl}/api/v1/examples`, { params: { numExamples: 1 } })
      .then(res => dispatch(loadExampleSuccess(res.data)))
      .then(() => dispatch(clearError()))
      .catch(err => dispatch(loadExampleFailure(err)));
  };
}

export function setContextText(text) {
  return (dispatch, getState) => {
    // Test if we have an error, if we do and the text has changed, we clear that error as user
    // is taking steps to fix.
    const { predictions } = getState();
    if (isErrorFixed(predictions.error, text, 'context')) {
      dispatch(clearError());
    }
    dispatch(setInputText(text, SET_CONTEXT_TEXT));
  };
}

export function setQueryText(text) {
  return (dispatch, getState) => {
    // Test if we have an error, if we do and the text has changed, we clear that error as user
    // is taking steps to fix.
    const { predictions } = getState();
    if (isErrorFixed(predictions.error, text, 'query')) {
      dispatch(clearError());
    }
    dispatch(setInputText(text, SET_QUERY_TEXT));
  };
}
