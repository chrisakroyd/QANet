import React from 'react';
import './introduction.scss';

const Introduction = () => (
  <div className="intro-container">
    <h2>QANet - Demo</h2>
    <h3>What is this thing?</h3>
    <p>
      This is a demo of the Machine Comprehension network QANet that answers natural language
      questions by pointing to the start and end words of an answer span from a section
      of relevant english language text.
    </p>
  </div>
);

export default Introduction;
