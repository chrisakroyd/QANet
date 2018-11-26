import {
  LOAD_TEXT, LOAD_TEXT_FAILURE, LOAD_TEXT_SUCCESS, SET_CONTEXT_TEXT,
  SET_CONTEXT_URL_TEXT, SET_QUERY_TEXT, ENTER_TEXT_MANUALLY,
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
    case ENTER_TEXT_MANUALLY:
      return Object.assign({}, state, {
        loadExample: false,
        context: '',
        query: '',
      });
    case SET_CONTEXT_URL_TEXT:
      return Object.assign({}, state, {
        contextUrl: action.text,
      });
    case LOAD_TEXT:
      return Object.assign({}, state, {
        loading: true,
        loadExample: true,
      });
    case LOAD_TEXT_SUCCESS:
      return Object.assign({}, state, {
        loading: false,
        query: action.query,
        context: action.context,
        error: null,
      });
    case LOAD_TEXT_FAILURE:
      return Object.assign({}, state, {
        loading: false,
        error: action.error,
      });
    default:
      return state;
  }
};

export default text;
