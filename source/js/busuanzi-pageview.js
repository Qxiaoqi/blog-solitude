(function () {
  'use strict'

  var pagePath = window.location.pathname
  var callbackName = 'BusuanziCallback_' + Math.floor(Math.random() * 1099511627776)
  var counterNames = ['site_pv', 'page_pv', 'site_uv']
  var request = document.createElement('script')

  function setCounter(name, value) {
    document.querySelectorAll('#busuanzi_value_' + name).forEach(function (element) {
      element.textContent = value
    })

    document.querySelectorAll('#busuanzi_container_' + name).forEach(function (element) {
      element.style.display = 'inline'
    })
  }

  function getPageviewBaseline(path) {
    var yearMatch = path.match(/^\/(\d{4})\//)

    // Non-article pages do not need a synthetic page-view baseline.
    if (!yearMatch) return 0

    var publishYear = Number(yearMatch[1])
    var yearsSince2017 = Math.max(0, publishYear - 2017)
    var min = 80 + yearsSince2017 * 70
    var max = 250 + yearsSince2017 * 120
    var hash = 2166136261

    // FNV-1a gives every path a stable, random-looking position in its year range.
    for (var i = 0; i < path.length; i++) {
      hash ^= path.charCodeAt(i)
      hash = Math.imul(hash, 16777619)
    }

    return min + ((hash >>> 0) % (max - min + 1))
  }

  function showFallback() {
    counterNames.forEach(function (name) {
      document.querySelectorAll('#busuanzi_value_' + name).forEach(function (element) {
        element.textContent = '--'
      })
    })
  }

  function cleanup() {
    if (request.parentNode) request.parentNode.removeChild(request)

    try {
      delete window[callbackName]
    } catch (error) {
      window[callbackName] = undefined
    }
  }

  window[callbackName] = function (data) {
    // Ignore a response from the previous page when PJAX navigation is very fast.
    if (window.location.pathname === pagePath) {
      var pagePv = Number(data.page_pv) || 0

      setCounter('site_pv', data.site_pv)
      setCounter('page_pv', getPageviewBaseline(pagePath) + pagePv)
      setCounter('site_uv', data.site_uv)
    }

    cleanup()
  }

  request.async = true
  // Busuanzi identifies page_pv from the Referer header. Modern browsers only
  // send the origin cross-site by default, so opt in to the full URL for this
  // request only instead of weakening the referrer policy for the whole site.
  request.referrerPolicy = 'unsafe-url'
  request.src =
    'https://busuanzi.ibruce.info/busuanzi?jsonpCallback=' +
    encodeURIComponent(callbackName)
  request.onerror = function () {
    if (window.location.pathname === pagePath) showFallback()
    cleanup()
  }

  document.head.appendChild(request)
})()
