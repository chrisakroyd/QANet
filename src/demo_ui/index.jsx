import React from 'react';
import createBrowserHistory from 'history/createBrowserHistory';
import thunk from 'redux-thunk';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';

import { getExample } from './actions/compositeActions';
import Root from './components/Root';
import { predictions, text } from './constants/defaults';
import searchApp from './reducers';

import './index.scss';

let middleware;

const history = createBrowserHistory();


if (process.env.NODE_ENV !== 'production') {
  const { logger } = require('redux-logger');
  const { composeWithDevTools } = require('redux-devtools-extension');
  middleware = composeWithDevTools(applyMiddleware(
    routerMiddleware(history),
    thunk,
    logger,
  ));
} else {
  middleware = applyMiddleware(routerMiddleware(history), thunk);
}

const reducers = combineReducers({
  ...searchApp,
  router: routerReducer,
});

const defaultState = { predictions, text };

const store = createStore(
  reducers,
  defaultState,
  middleware,
);

store.dispatch(getExample());

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);
