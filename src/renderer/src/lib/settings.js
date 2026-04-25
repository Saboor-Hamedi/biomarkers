export const saveSetting = (key, value) => {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (error) {
    console.error('Failed to save setting:', error);
    return false;
  }
};

export const getSetting = (key, defaultValue = '') => {
  try {
    return localStorage.getItem(key) || defaultValue;
  } catch (error) {
    console.error('Failed to read setting:', error);
    return defaultValue;
  }
};
