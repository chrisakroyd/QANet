import React from 'react';
import PropTypes from 'prop-types';

import './input-box.scss';

class InputBox extends React.Component {
  render() {
    return (
      <div className="input-box">
        <textarea
          className="text-area"
          value={this.props.value}
          onChange={() => this.props.onKeyPress(this.textInput.value)}
          ref={(input) => { this.textInput = input; }}
        />
      </div>
    );
  }
}

InputBox.propTypes = {
  value: PropTypes.string.isRequired,
  onKeyPress: PropTypes.func.isRequired,
};

export default InputBox;
