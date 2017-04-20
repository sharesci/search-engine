
function logoutAction(req, res) {
	delete req.session['user_id'];
	req.session.destroy();
	if (req.body.successRedirect) {
		res.redirect(req.body.successRedirect);
	} else {
		res.redirect('/');
	}
}

module.exports = {
	logoutAction: logoutAction,
};


