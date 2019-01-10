import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import './inputs.scss';


const Button = ({ onClick, label, enabled }) => (
  <div
    className={classNames('button', 'general-button', { enabled })}
    onClick={() => onClick()}
    onKeyPress={() => onClick()}
    role="button"
    tabIndex={0}
  >
    {label}
  </div>
);

Button.propTypes = {
  onClick: PropTypes.func.isRequired,
  label: PropTypes.string.isRequired,
  enabled: PropTypes.bool,
};

Button.defaultProps = {
  enabled: true,
};

export default Button;
