import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import InputBar from './InputBar/InputBar';
import PredictButton from './PredictButton/PredictButton';
import WordHeat from './WordHeat/WordHeat';
import predictShape from './../../prop-shapes/predictShape';
import textShape from './../../prop-shapes/textShape';
// import { predictShape, textShape } from './../../prop-shapes';
import './demo.scss';

const Demo = ({
  predict, setQueryText, setContextText, setContextUrl, text, predictions,
}) => {
  return (
    <div className="demo-body">
      <div className="content">
        <div className="row">
          <div className="step-container"></div>
          <div className="intro-container">
            <h2>QANet - Demo</h2>
            <h3>What is this thing?</h3>
            <p>This demo lets you   </p>
          </div>
        </div>
        <div className="row">
          <div className="step-container">
            <div className="step">
              1 / 3
            </div>
            <div className="label">
              Ask a question.
            </div>
          </div>
          <InputBar
            onEnter={predict}
            value={text.query}
            onKeyPress={setQueryText}
          />
        </div>

        <div className="row">
          <div className="step-container">
            <div className="step">
              2 / 3
            </div>
            <div className="label">
              Enter a context.
            </div>
          </div>

          <div className="context-options">
            <div className="context-option active">
              Enter text manually
            </div>
            <div className="context-or">Or</div>
            <div className="context-option">
              Load text from URL.
            </div>
          </div>

          {/*<InputBar*/}
            {/*onEnter={predict}*/}
            {/*value={text.query}*/}
            {/*onKeyPress={setQueryText}*/}
          {/*/>*/}
        </div>

        <div className="row">
          <div className="step-container">
            <div className="step">
              3 / 3
            </div>
            <div className="label">
              Get an Answer.
            </div>
          </div>
          <div className="button-container">
            <PredictButton onEnter={predict} />
          </div>
        </div>
      </div>
    </div>
  );
};

Demo.propTypes = {
  // Functions
  predict: PropTypes.func.isRequired,
  setQueryText: PropTypes.func.isRequired,
  setContextText: PropTypes.func.isRequired,
  setContextUrl: PropTypes.func.isRequired,
  // Data
  text: textShape.isRequired,
  predictions: predictShape.isRequired,
};


export default Demo;
