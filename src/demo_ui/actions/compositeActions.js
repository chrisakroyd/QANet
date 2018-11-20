import axios from 'axios';

import config from '../config';
import { loadUrl, loadUrlSuccess, loadUrlFailure } from './textActions'
import { predict, predictSuccess, predictFailure } from './predictActions';


export function getPrediction() {
  return (dispatch, getState) => {
    const { text } = getState();

    dispatch(predict());
    // return axios.post(`${config.siteUrl}/api/v1/tweets/predict`, {textShape: ''})
    return axios.post(`http://localhost:5000'/qanet/predict'`, { context: text.context, query: text.query })
      .then(res => dispatch(predictSuccess(res.data)))
      .catch(err => dispatch(predictFailure(err)));
  };
}

export function getContextFromUrl() {
  return (dispatch, getState) => {
    const { text } = getState();

    dispatch(loadUrl());
    // return axios.post(`${config.siteUrl}/api/v1/tweets/predict`, {textShape: ''})
    return axios.post(`http://localhost:5000/text/extract`, { url: text.contextUrl })
      .then(res => dispatch(loadUrlSuccess(res.data)))
      .catch(err => dispatch(loadUrlFailure(err)));
  };
}
