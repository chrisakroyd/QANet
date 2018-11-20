import {
  LOAD_TEXT, LOAD_TEXT_FAILURE, LOAD_TEXT_SUCCESS, SET_CONTEXT_TEXT,
  SET_CONTEXT_URL_TEXT, SET_QUERY_TEXT, SET_URL_LOAD_FLAG,
} from '../constants/actions';

const text = (state = {}, action) => {
  switch (action.type) {
    case SET_CONTEXT_TEXT:
      return Object.assign({}, state, {
        context: action.text,
      });
    case SET_QUERY_TEXT:
      return Object.assign({}, state, {
        query: action.text,
      });
    case SET_URL_LOAD_FLAG:
      return Object.assign({}, state, {
        loadContextFromUrl: action.loadFromUrl,
      });
    case SET_CONTEXT_URL_TEXT:
      return Object.assign({}, state, {
        contextUrl: action.text,
      });
    case LOAD_TEXT:
      return Object.assign({}, state, {
        loading: true,
      });
    case LOAD_TEXT_SUCCESS:
      return Object.assign({}, state, {
        loading: false,
        context: action.text,
        errorCode: -1,
      });
    case LOAD_TEXT_FAILURE:
      return Object.assign({}, state, {
        loading: false,
        errorCode: action.errorCode,
      });
    default:
      return state;
  }
};

export default text;
