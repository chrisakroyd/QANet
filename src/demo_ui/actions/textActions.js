import {
  LOAD_TEXT, LOAD_TEXT_SUCCESS, LOAD_TEXT_FAILURE,
  ENTER_TEXT_MANUALLY, SET_QUERY_TEXT,
} from '../constants/actions';

export function setInputText(text, type = SET_QUERY_TEXT) {
  return {
    type,
    text,
  };
}

export function enterTextManually() {
  return {
    type: ENTER_TEXT_MANUALLY,
  };
}

export function loadExample() {
  return {
    type: LOAD_TEXT,
  };
}

export function loadExampleSuccess(data) {
  const lastExample = data.data[data.numExamples - 1];
  return {
    type: LOAD_TEXT_SUCCESS,
    query: lastExample.query,
    context: lastExample.context,
  };
}

export function loadExampleFailure(error) {
  return {
    type: LOAD_TEXT_FAILURE,
    error,
  };
}
