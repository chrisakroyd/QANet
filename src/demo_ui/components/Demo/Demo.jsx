import React from 'react';
import { Route } from 'react-router';
import PropTypes from 'prop-types';

import Introduction from './Introduction/Introduction';
import InputPage from '../Inputs/InputPage';
import ResultPage from '../Results/ResultsPage';
import predictShape from './../../prop-shapes/predictShape';
import textShape from './../../prop-shapes/textShape';
// import { predictShape, textShape } from './../../prop-shapes';
import './demo.scss';

const Demo = ({
  predict, returnHome, setQueryText, setContextText, enterText,
  loadExample, text, predictions,
}) => (
  <div className="demo-body">
    <div className="section">
      <Introduction />
    </div>
    <Route
      exact
      path="/"
      render={() => (
        <InputPage
          predict={predict}
          setQueryText={setQueryText}
          setContextText={setContextText}
          enterText={enterText}
          loadExample={loadExample}
          predictions={predictions}
          text={text}
        />
      )}
    />
    <Route
      exact
      path="/results"
      render={() => (
        <ResultPage
          goBack={returnHome}
          query={text.query}
          predictions={predictions}
        />
      )}
    />
  </div>
);

Demo.propTypes = {
  // Functions
  predict: PropTypes.func.isRequired,
  returnHome: PropTypes.func.isRequired,
  setQueryText: PropTypes.func.isRequired,
  setContextText: PropTypes.func.isRequired,
  enterText: PropTypes.func.isRequired,
  loadExample: PropTypes.func.isRequired,
  // Data
  text: textShape.isRequired,
  predictions: predictShape.isRequired,
};

export default Demo;
