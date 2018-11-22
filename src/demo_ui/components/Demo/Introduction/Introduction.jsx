import React from 'react';
import './introduction.scss';

const Introduction = () => (
  <div className="intro-container">
    <h2>QANet - Demo</h2>
    <h3>What is this thing?</h3>
    <p>
      This is a demo of the machine-comprehension network QANet that extracts an answer
      from an arbitrary section of english language text. Due to how the network is trained,
      we expect a correct answer exists within the text.
    </p>
  </div>
);

export default Introduction;
