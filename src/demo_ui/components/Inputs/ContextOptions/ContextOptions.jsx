import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './context-options.scss';

const ContextOptions = ({ enterTextFunc, exampleFunc, loadExample }) => {
  const manualClass = classNames('context-option', { active: !loadExample });
  const getUrlClass = classNames('context-option', { active: loadExample });

  return (
    <div className="context-options">
      <div className={manualClass} onClick={() => enterTextFunc()} onKeyPress={() => enterTextFunc()} role="button" tabIndex={0}>
        Enter text manually
      </div>
      <div className="context-or">Or</div>
      <div className={getUrlClass} onClick={() => exampleFunc()} onKeyPress={() => exampleFunc()} role="button" tabIndex={0}>
        Load a random example
      </div>
    </div>);
};

ContextOptions.propTypes = {
  enterTextFunc: PropTypes.func.isRequired,
  exampleFunc: PropTypes.func.isRequired,
  loadExample: PropTypes.bool.isRequired,
};

export default ContextOptions;
