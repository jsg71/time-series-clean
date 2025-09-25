// MathJax v3 local config for MkDocs Material + Arithmatex (offline)
window.MathJax = {
  loader: {
    load: ['[tex]/ams', '[tex]/boldsymbol', '[tex]/textmacros']
  },
  tex: {
    packages: {'[+]': ['ams', 'boldsymbol', 'textmacros']},
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
    processEnvironments: true
  },
  chtml: {
    // If you serve MkDocs at site root, keep the leading slash:
    fontURL: '/assets/mathjax/es5/output/chtml/fonts/woff-v2'
    // If you serve under a subpath, switch to the relative form:
    // fontURL: 'assets/mathjax/es5/output/chtml/fonts/woff-v2'
  },
  options: {
    processHtmlClass: 'arithmatex',
    ignoreHtmlClass: '.*'
  },
  // Let us decide when to typeset; we'll do it in startup.ready
  startup: {
    typeset: false,
    // Runs AFTER MathJax is fully loaded (safe place to typeset)
    ready: () => {
      MathJax.startup.defaultReady();  // keep MJ's normal initialization
      MathJax.typesetPromise();        // initial typeset on first load
    }
  }
};

// Re-typeset on MkDocs Material SPA page changes
if (window.document$) {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
}
