import { User } from '../../entities/user.entity.js'

export class UserInfoValidatonService {

    is_valid_username(username: string): error {
        if (!username) {
            return { errno: 1, errstr: "Empty Username" };
        }
        if (username.length < 4) {
            return { errno: 1, errstr: "username should be atleast 4 characters long." }
        }
        return { errno: 0, errstr: "" };
    }

    is_valid_email(email: string): error {
        //regex is taken from http://stackoverflow.com/questions/46155/validate-email-address-in-javascript
        if (!email) {
            return { errno: 1, errstr: "Empty Email" };
        }
        var re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
        if(!re.test(email)) {
            return { errno: 1, errstr: "Invalid email format" };
        }
        return { errno: 0, errstr: "" };
    }

    do_emails_match(email: string, conf_email: string): error {
        if (email != conf_email) {
            return { errno: 1, errstr: "Emails do not match." };
        }
        return { errno: 0, errstr: "" };
    }

    is_valid_password(password: string): error {
        if (!password) {
            return { errno: 1, errstr: "Empty password." };
        }
        if(password.length < 8) {
            return { errno: 1, errstr: "Password should be atleast 8 characters long." };
        }
        return { errno: 0, errstr: "" };
    }

    do_passwords_match(password: string, conf_password: string): error {
        if (password != conf_password) {
            return { errno: 1, errstr: "Passwords do not match." };
        }
        return { errno: 0, errstr: "" };
    }

    is_valid_firstname(firstname: string): error {
        if (!firstname) {
            return { errno: 1, errstr: "firstname cannot be empty."}
        }
        return { errno: 0, errstr: ""};
    }

    is_valid_lastname(lastname: string): error {
        if (!lastname) {
            return { errno: 1, errstr: "lastname cannot be empty" };
        }
        if (lastname.length < 2) {
            return { errno: 1, errstr: "lastname should be atleast 2 characters long." };
        }
        return { errno: 0, errstr: "" };
    }

    is_valid_self_bio(self_bio: string): error {
        return { errno: 0, errstr: ""};
    }

    is_valid_institution(institution: string): error {
        return { errno: 0, errstr: ""};
    }
}

export interface error {
    errno: number, //0 valid, 1 invalid
    errstr: string
}
