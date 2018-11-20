import React from 'react';
import PropTypes from 'prop-types';
import './search-button.scss';

const SearchButton = ({ onEnter }) => (
  <div className="search-button" onClick={() => onEnter()}>
    Process
  </div>
);

SearchButton.propTypes = {
  onEnter: PropTypes.func.isRequired,
};

export default SearchButton;
