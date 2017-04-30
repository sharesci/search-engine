
function is_valid_articleMetaJson(articleMetaJson) {
	if(!articleMetaJson) {
		return false;
	}
	try {
		var a = JSON.parse(articleMetaJson);
		if(!a || typeof a !== 'object') {
			return false;
		}
	}
	catch(err) {
		return false;
	}
	return true;
}

function is_valid_email(email) {
	if (!email) {
		return false;
	}
	return (typeof email === 'string' && (/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i).test(email));
}


function is_valid_username(username) {
	if (!username) {
		return false;
	}
	return (typeof username === 'string' && 4 <= username.length);
}

function is_valid_password(password) {
	if (!password) {
		return false;
	}
	return (typeof password === 'string' && 8 <= password.length);
}

function is_valid_firstname(firstname) {
	if (!firstname) {
		return false;
	}
	return (typeof firstname === 'string' && 1 <= firstname.length);
}

function is_valid_lastname(lastname) {
	if (!lastname) {
		return false;
	}
	return (typeof lastname === 'string' && 2 <= lastname.length);
}

function is_valid_self_bio(bio) {
	return true;
}

function is_valid_institution(institution) {
	return true;
}


module.exports = {
	is_valid_articleMetaJson:       is_valid_articleMetaJson,
	is_valid_email:                 is_valid_email,
	is_valid_username:              is_valid_username,
	is_valid_password:              is_valid_password,
	is_valid_firstname:             is_valid_firstname,
	is_valid_lastname:              is_valid_lastname,
	is_valid_self_bio:              is_valid_self_bio,
	is_valid_institution:           is_valid_institution
};

