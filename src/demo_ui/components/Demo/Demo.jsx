import React from 'react';
import { Route } from 'react-router';
import PropTypes from 'prop-types';

import Introduction from './Introduction/Introduction';
import InputPage from '../Inputs/InputPage';
import predictShape from './../../prop-shapes/predictShape';
import textShape from './../../prop-shapes/textShape';
// import { predictShape, textShape } from './../../prop-shapes';
import './demo.scss';

const Demo = ({
  predict, setQueryText, setContextText, setContextUrlText, setContextUrlFlag, text, predictions,
}) => (
  <div className="demo-body">
    <div className="section">
      <Introduction />
    </div>
    <InputPage
      predict={predict}
      setQueryText={setQueryText}
      setContextText={setContextText}
      setContextUrlText={setContextUrlText}
      setContextUrlFlag={setContextUrlFlag}
      text={text}
    />
  </div>
);

Demo.propTypes = {
  // Functions
  predict: PropTypes.func.isRequired,
  setQueryText: PropTypes.func.isRequired,
  setContextText: PropTypes.func.isRequired,
  setContextUrlText: PropTypes.func.isRequired,
  setContextUrlFlag: PropTypes.func.isRequired,
  // Data
  text: textShape.isRequired,
  predictions: predictShape.isRequired,
};

export default Demo;
