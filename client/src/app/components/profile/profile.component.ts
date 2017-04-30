import { Component, OnInit, Input } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { SharedService } from '../../services/shared.service.js';
import { AccountService } from '../../services/account.service.js';
import { User, IUserWrapper } from '../../entities/user.entity.js';

@Component({
    templateUrl: 'src/app/components/profile/profile.component.html',
    styleUrls: ['src/app/components/profile/profile.component.css']
})

export class ProfileComponent implements OnInit {
    user = new User();
    emails: string[] = [];
    emailsupdated: string[] = [];
    err = { profile : "", account: ""};
    password = { old: "", new: "", conf: "" };

    constructor(private _sharedService: SharedService, private _accountService: AccountService,
        private _route: ActivatedRoute) {
    }

    ngOnInit() {
        this._accountService.getUserInfo(this._route.snapshot.params['username'])
            .map(response => <IUserWrapper>response)
            .subscribe(
            data => this.showUserInfo(data),
            error => console.log(error)
            )

        this._accountService.getUserEmail(this._route.snapshot.params['username'])
            .subscribe(
            data => { this.emails = this.emails.concat(data.emails);
                      this.emailsupdated = this.emailsupdated.concat(data.emails);
                    },
            error => this.err.account = error
            )
    }

    showUserInfo(userWrapper: IUserWrapper) {
        if (userWrapper.errno == 0) {
            this.user = userWrapper.userJson;
        }
    }

    saveUserInfo() {
        this._accountService.updateUserInfo(this.user)
            .subscribe(
                results => { },
                error => { this.err.profile = error }
            )
    }

    addEmptyRow() {
        this.emails = this.emails.concat([""]);
        this.emailsupdated = this.emailsupdated.concat([""]);
    }

    deleteEmail(index: number) {
        this._accountService.deleteUserEmail(this.user.username, this.emailsupdated[index])
            .subscribe(
            result => { this.err.account = "Successfully deleted"; 
                        this.emailsupdated.splice(index, 1); 
                        this.emails = this.emailsupdated },
            error => { this.err.account = error }
            )
    }

    saveEmail(index: number) {
        if(this.emails[index]) {
            this._accountService.deleteUserEmail(this.user.username, this.emails[index])
                .subscribe(
                result => { } ,
                error => { this.err.account = error; return }
                )
        }
        this._accountService.addUserEmail(this.user.username, this.emailsupdated[index])
            .subscribe(
            result => { this.err.account = "Success updated"; this.emails = this.emailsupdated },
            error => this.err.account = error
            )
    }

    updateEmails(index: number) {
        this.emailsupdated[index] = $("#email" + index).val();
    }

    savePassword() {
        if(this.password.new != this.password.conf) {
            this.err.account = "Passwords do not match";
            return;
        }

        this._accountService.updateUserPassword(this.user.username, this.password.old, this.password.new)
            .subscribe(
                results => { this.err.account = "Success" },
                error => { this.err.account = error }
            )
    }
}