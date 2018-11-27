import axios from 'axios';

import config from '../config';
import { loadExample, loadExampleSuccess, loadExampleFailure } from './textActions';
import { predict, predictSuccess, predictFailure } from './predictActions';

const demoUrl = `http://localhost:${config.demoPort}`;

export function runPrediction() {
  return (dispatch, getState) => {
    const { text } = getState();

    dispatch(predict());
    return axios.post(`${demoUrl}/api/v1/qanet/predict`, { context: text.context, query: text.query })
      .then(res => dispatch(predictSuccess(res.data)))
      .catch(err => dispatch(predictFailure(err)));
  };
}

export function getExample() {
  return (dispatch) => {
    dispatch(loadExample());
    return axios.get(`${demoUrl}/api/v1/examples`, { params: { numExamples: 1 } })
      .then(res => dispatch(loadExampleSuccess(res.data)))
      .catch(err => dispatch(loadExampleFailure(err)));
  };
}
