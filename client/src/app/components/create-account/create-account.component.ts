import { Component } from '@angular/core';
import { AccountService } from '../../services/account.service.js';

@Component({
    selector: 'ss-account',
    templateUrl: 'src/app/components/create-account/create-account.component.html',
})

export class CreateAccountComponent{
    username: string;
    password: string;
    errstr: string;

    constructor(private _accountService: AccountService) { }

    createAccount() {
        this._accountService.create(this.username, this.password)
            .subscribe(
                result => this.handleResult(result),
                error => console.log(error)
            );
    }

    handleResult(result: any) {
        if (result.errno == '0') {
            this.errstr = "Account created. You can login now.";
        }
        else {
            this.errstr = result.errstr;
        }
    }
}