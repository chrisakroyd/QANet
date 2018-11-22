import React from 'react';
import PropTypes from 'prop-types';

import './step.scss';

const Step = ({ number, label }) => (
  <div className="step-container">
    <div className="step">{number}</div>
    <div className="label">{label}</div>
  </div>
);

Step.propTypes = {
  number: PropTypes.number.isRequired,
  label: PropTypes.string.isRequired,
};

export default Step;
