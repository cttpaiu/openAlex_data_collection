/**
 * OpenAlex CLI Documentation - Main JavaScript
 * Handles mobile navigation and UI interactions
 */

document.addEventListener('DOMContentLoaded', function() {
  initMobileMenu();
  initTableWrappers();
});

/**
 * Mobile menu toggle
 */
function initMobileMenu() {
  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');
  const main = document.querySelector('.main');

  if (!menuToggle || !sidebar) return;

  menuToggle.addEventListener('click', function() {
    menuToggle.classList.toggle('active');
    sidebar.classList.toggle('active');
    
    // Prevent body scroll when menu is open
    if (sidebar.classList.contains('active')) {
      main.style.pointerEvents = 'none';
    } else {
      main.style.pointerEvents = 'auto';
    }
  });

  // Close menu when clicking outside
  document.addEventListener('click', function(e) {
    if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
      menuToggle.classList.remove('active');
      sidebar.classList.remove('active');
      main.style.pointerEvents = 'auto';
    }
  });

  // Close menu on escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      menuToggle.classList.remove('active');
      sidebar.classList.remove('active');
      main.style.pointerEvents = 'auto';
    }
  });
}

/**
 * Wrap tables in scrollable containers
 */
function initTableWrappers() {
  const tables = document.querySelectorAll('table');
  
  tables.forEach(function(table) {
    // Skip if already wrapped
    if (table.parentElement.classList.contains('table-wrapper')) return;
    
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';
    
    // Insert wrapper before table
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
}
