import React from 'react';
import PropTypes from 'prop-types';
import './buttons.scss';

const Button = ({ onClick, label }) => (
  <div className="button" onClick={() => onClick()} onKeyPress={() => onClick()} role="button" tabIndex={0}>
    {label}
  </div>
);

Button.propTypes = {
  onClick: PropTypes.func.isRequired,
  label: PropTypes.string.isRequired,
};

export default Button;
