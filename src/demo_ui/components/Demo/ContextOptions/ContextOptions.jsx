import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './context-options.scss';

const ContextOptions = ({ onClick, useUrl }) => {
  const manualClass = classNames('context-option', { active: !useUrl });
  const getUrlClass = classNames('context-option', { active: useUrl });

  return (
    <div className="context-options">
      <div className={manualClass} onClick={() => onClick(false)} onKeyPress={() => onClick(false)} role="button" tabIndex={0}>
        Enter text manually
      </div>
      <div className="context-or">Or</div>
      <div className={getUrlClass} onClick={() => onClick(true)} onKeyPress={() => onClick(true)} role="button" tabIndex={0}>
        Load text from URL.
      </div>
    </div>);
};

ContextOptions.propTypes = {
  onClick: PropTypes.func.isRequired,
  useUrl: PropTypes.bool.isRequired,
};

export default ContextOptions;
