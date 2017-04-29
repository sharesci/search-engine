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
    constructor(private _sharedService: SharedService, private _accountService: AccountService,
                private _route: ActivatedRoute) { 
    }

    user = new User();

    ngOnInit() {
        this._accountService.getUserInfo(this._route.snapshot.params['username'])
            .map(response => <IUserWrapper>response)
            .subscribe(
                data => this.showUserInfo(data) ,
                error => console.log(error)
        )
    }

    showUserInfo(userWrapper: IUserWrapper) {
        if (userWrapper.errno == 0) {
            this.user = userWrapper.userJson;
            console.log(this.user);
        }
    }
}