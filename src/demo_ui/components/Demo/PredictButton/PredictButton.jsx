import React from 'react';
import PropTypes from 'prop-types';
import './predict-button.scss';

const PredictButton = ({ onEnter }) => (
  <div className="predict-button" onClick={() => onEnter()} onKeyPress={() => onEnter()} role="button" tabIndex={0}>
    Process
  </div>
);

PredictButton.propTypes = {
  onEnter: PropTypes.func.isRequired,
};

export default PredictButton;
