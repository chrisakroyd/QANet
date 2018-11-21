import React from 'react';
import PropTypes from 'prop-types';
import './predict-button.scss';

const PredictButton = ({ onEnter }) => (
  <div className="predict-button" onClick={() => onEnter()}>
    Process
  </div>
);

PredictButton.propTypes = {
  onEnter: PropTypes.func.isRequired,
};

export default PredictButton;
