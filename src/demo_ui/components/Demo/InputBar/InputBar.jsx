import React from 'react';
import PropTypes from 'prop-types';

import './input-bar.scss';

class InputBar extends React.Component {
  render() {
    return (
      <div className="input-bar">
        <div className="input-container">
          <input
            onChange={() => this.props.onKeyPress(this.textInput.value)}
            placeholder={this.props.placeholder}
            value={this.props.value}
            ref={(input) => { this.textInput = input; }}
          />
        </div>
      </div>
    );
  }
}

InputBar.propTypes = {
  onKeyPress: PropTypes.func.isRequired,
  placeholder: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
};

export default InputBar;
