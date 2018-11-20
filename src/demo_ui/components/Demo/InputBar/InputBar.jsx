import React from 'react';
import PropTypes from 'prop-types';

import './input-bar.scss';

class SearchBar extends React.Component {
  render() {
    const searchPlaceholder = 'What textShape do you want to analyse?';

    return (
      <div className="search-bar">
        <div className="search-input-container">
          <input
            className="search-input"
            onChange={() => this.props.onKeyPress(this.textInput.value)}
            onKeyPress={(event) => { if (event.key === 'Enter') this.props.onEnter(); }}
            placeholder={searchPlaceholder}
            value={this.props.value}
            ref={(input) => { this.textInput = input; }}
          />
        </div>
      </div>
    );
  }
}

SearchBar.propTypes = {
  onEnter: PropTypes.func.isRequired,
  value: PropTypes.string.isRequired,
  onKeyPress: PropTypes.func.isRequired,
};

export default SearchBar;
