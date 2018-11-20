import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import InputBar from './InputBar/InputBar';
import SearchButton from './SearchButton/SearchButton';
import WordHeat from './WordHeat/WordHeat';
import predictShape from './../../prop-shapes/predictShape';
import textShape from './../../prop-shapes/textShape';
// import { predictShape, textShape } from './../../prop-shapes';
import './demo.scss';

const Demo = ({
  predict, setQueryText, setContextText, setContextUrl, text, predictions,
}) => {
  return (
    <div className="live dash-body">
      <div className="body-header">
        <h1>Demo</h1>
      </div>

      <div className="tile-row">
        <div className="tile large-tile">
          <div className="tile-header">
            <h3>Demo</h3>
          </div>
          <div className="tile-body">
            <h4>1. Enter Query</h4>
            <div className="enter-text-row">
              <InputBar
                onEnter={predict}
                value={text.query}
                onKeyPress={setQueryText}
              />
            </div>
          </div>
          <div className="tile-body">
            <h4>1. Enter Query</h4>
            <div className="enter-text-row">
              <InputBar
                onEnter={predict}
                value={text.context}
                onKeyPress={setContextText}
              />
            </div>
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
