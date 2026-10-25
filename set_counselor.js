// Grants the "counselor" permission to one Google account so the dashboard can read check-ins.
// Usage:  node set_counselor.js your-email@gmail.com
// The account must have signed in to the dashboard at least once.
// After running, sign out and back in on the dashboard.

const { initializeApp, cert } = require("firebase-admin/app");
const { getAuth } = require("firebase-admin/auth");
const serviceAccount = require("./serviceAccountKey.json");

initializeApp({ credential: cert(serviceAccount) });

const email = process.argv[2];
if (!email) {
  console.error("Usage: node set_counselor.js your-email@gmail.com");
  process.exit(1);
}

const auth = getAuth();
auth.getUserByEmail(email)
  .then(user => auth.setCustomUserClaims(user.uid, { counselor: true }))
  .then(() => console.log("Counselor access granted to " + email + ". Sign out and back in on the dashboard to apply it."))
  .catch(err => {
    console.error("Failed: " + err.message);
    process.exit(1);
  });
