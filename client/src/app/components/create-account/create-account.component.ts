import { Component } from '@angular/core';
import { User } from '../../entities/user.entity.js';
import { AccountService } from '../../services/account.service.js';
import { AuthenticationService } from '../../services/authentication.service.js';
import { UserInfoValidatonService, error } from './user-info-validaton.service.js';

@Component({
    selector: 'ss-account',
    templateUrl: 'src/app/components/create-account/create-account.component.html',
    providers: [UserInfoValidatonService]
})

export class CreateAccountComponent {
    user = new User();
    err = { email: "", main: "", password: "", self_bio: "", institution: "", firstname: "", lastname: "", username: "", conf_email: "", conf_password: "" };
    msgclass = "alert alert-success";
    formValid: boolean = true;

    constructor(private _accountService: AccountService, private _authService: AuthenticationService,
                private _validationService: UserInfoValidatonService) { }

    createAccount() {
        if (this.formValid) {
            this._accountService.create(this.user)
                .subscribe(
                result => this.handleResult(result),
                error => console.log(error)
                );
        }
        else {
            this.err.main = "Fill all required fields and resolve all the errors.";
            this.msgclass = "alert alert-danger";
        }
    }

    handleResult(result: any) {
        if (result.errno == '0') {
            this._authService.login(this.user.username, this.user.password)
                .subscribe(
                result => {
                    this._accountService.addUserEmail(this.user.username, $("#email").val());
                    this.err.main = "Account created. You can login now.";
                    this.msgclass = "alert alert-success";
                },
                error => { return }
                )
        }
        else {
            this.err.main = result.errstr;
            this.msgclass = "alert alert-danger";
        }
    }

    validate(event: any) {
        let id = event.srcElement.id;
        let currentErr : error = { errno: 0, errstr: "" };
        console.log(id);
        switch (id) {
            case "email":
                     currentErr = this._validationService.is_valid_email($("#email").val());
                     this.err.email = currentErr.errstr;
                     break;
            case "conf_email":
                    currentErr = this._validationService.do_emails_match($("#email").val(), $("#conf_email").val());
                    this.err.conf_email = currentErr.errstr;
                    break;
            case "password":
                    currentErr = this._validationService.is_valid_password(this.user.password);
                    this.err.password = currentErr.errstr;
                    break;
            case "conf_password":
                    currentErr = this._validationService.do_passwords_match(this.user.password, $("#conf_password").val());
                    this.err.conf_password = currentErr.errstr;
                    break;
            case "self_bio":
                    currentErr = this._validationService.is_valid_self_bio(this.user.self_bio);
                    this.err.self_bio = currentErr.errstr;
                    break;
            case "institution":
                    currentErr = this._validationService.is_valid_institution(this.user.institution);
                    this.err.institution = currentErr.errstr;
                    break;
            case "firstname":
                    currentErr = this._validationService.is_valid_firstname(this.user.firstname);
                    this.err.firstname = currentErr.errstr;
                    break;
            case "lastname":
                    currentErr = this._validationService.is_valid_lastname(this.user.lastname);
                    this.err.lastname = currentErr.errstr;
                    break;
            case "username":
                    currentErr = this._validationService.is_valid_username(this.user.username);
                    this.err.username = currentErr.errstr;
                    break;
        }
        this.formValid = Boolean(currentErr.errno);
    }

    matchEmail(event: any) {
        console.log(event);
        let email = $("#email").val();
        let conf_email = $("#conf_email").val();
        if (email != conf_email) {
            this.err.email = "Emails do not match.";
            this.formValid = false;
        }
        else {
            this.err.email = "";
            this.formValid = true;
        }
    }

    matchPassword() {
        let password = $("#password").val();
        let conf_password = $("#conf_password").val();
        if (password != conf_password) {
            this.err.password = "Passwords do not match.";
            this.formValid = false;
        }
        else {
            this.err.password = "";
            this.formValid = true;
        }
    }
}