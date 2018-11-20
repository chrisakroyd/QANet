import {
  LOAD_TEXT, LOAD_TEXT_SUCCESS, LOAD_TEXT_FAILURE,
  SET_URL_LOAD_FLAG, SET_QUERY_TEXT,
} from '../constants/actions';

export function setInputText(text, type = SET_QUERY_TEXT) {
  return {
    type,
    text,
  };
}

export function setURLFlag(flag) {
  return {
    type: SET_URL_LOAD_FLAG,
    loadFromUrl: flag,
  };
}

export function loadUrl() {
  return {
    type: LOAD_TEXT,
  };
}

export function loadUrlSuccess(data) {
  return {
    type: LOAD_TEXT_SUCCESS,
    data,
  };
}

export function loadUrlFailure(errorCode) {
  return {
    type: LOAD_TEXT_FAILURE,
    errorCode,
  };
}
